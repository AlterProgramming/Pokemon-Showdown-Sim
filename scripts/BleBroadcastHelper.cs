using System;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading;
using Windows.Devices.Bluetooth.Advertisement;

public static class BleBroadcastHelper
{
    public static int Main(string[] args)
    {
        try
        {
            var options = BroadcastOptions.Parse(args);
            var payload = HexToBytes(options.PayloadHex);
            var advertisement = new BluetoothLEAdvertisement();
            advertisement.ManufacturerData.Add(
                new BluetoothLEManufacturerData(options.CompanyId, payload.AsBuffer())
            );

            var publisher = new BluetoothLEAdvertisementPublisher(advertisement);
            publisher.Start();
            Thread.Sleep(500);

            if (publisher.Status != BluetoothLEAdvertisementPublisherStatus.Started)
            {
                Console.Error.WriteLine("BLE advertisement did not start. Publisher status: " + publisher.Status);
                publisher.Stop();
                return 1;
            }

            Console.WriteLine("Started");
            Thread.Sleep(options.DurationMilliseconds);
            publisher.Stop();
            Console.WriteLine("Stopped");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
    }

    private static byte[] HexToBytes(string hex)
    {
        if (string.IsNullOrWhiteSpace(hex) || (hex.Length % 2) != 0)
        {
            throw new ArgumentException("Payload hex must be a non-empty even-length string.");
        }

        var bytes = new byte[hex.Length / 2];
        for (var i = 0; i < bytes.Length; i++)
        {
            bytes[i] = Convert.ToByte(hex.Substring(i * 2, 2), 16);
        }

        return bytes;
    }

    private sealed class BroadcastOptions
    {
        public ushort CompanyId { get; private set; }
        public string PayloadHex { get; private set; }
        public int DurationMilliseconds { get; private set; }

        public static BroadcastOptions Parse(string[] args)
        {
            var options = new BroadcastOptions
            {
                CompanyId = 65534,
                DurationMilliseconds = 120000,
                PayloadHex = string.Empty,
            };

            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                switch (arg)
                {
                    case "--company-id":
                        options.CompanyId = ushort.Parse(NextValue(args, ref i, arg));
                        break;
                    case "--payload-hex":
                        options.PayloadHex = NextValue(args, ref i, arg);
                        break;
                    case "--duration-ms":
                        options.DurationMilliseconds = int.Parse(NextValue(args, ref i, arg));
                        break;
                    default:
                        throw new ArgumentException("Unknown argument: " + arg);
                }
            }

            if (string.IsNullOrWhiteSpace(options.PayloadHex))
            {
                throw new ArgumentException("Missing required argument: --payload-hex");
            }

            if (options.DurationMilliseconds < 0)
            {
                throw new ArgumentException("Duration must be non-negative.");
            }

            return options;
        }

        private static string NextValue(string[] args, ref int index, string argName)
        {
            if (index + 1 >= args.Length)
            {
                throw new ArgumentException("Missing value for " + argName);
            }

            index += 1;
            return args[index];
        }
    }
}
